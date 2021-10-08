import wf_core_data.rosters.shared_constants
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

NWEA_TARGET_COLUMN_NAMES = [
    'School State Code',
    'School Name',
    'Previous Instructor ID',
    'Instructor ID',
    'Instructor State ID',
    'Instructor Last Name',
    'Instructor First Name',
    'Instructor Middle Initial',
    'User Name',
    'Email Address',
    'Class Name',
    'Subject',
    'Previous Student ID',
    'Student ID',
    'Student State ID',
    'Student Last Name',
    'Student First Name',
    'Student Middle Initial',
    'Student Date Of Birth',
    'Student Gender',
    'Student Grade',
    'Student Ethnic Group Name',
    'Student User Name',
    'Student Email'
]

NWEA_GENDER_MAP = {
    'M': 'M',
    'F': 'F',
    'unmatched_value': 'F',
    'na_value': 'F'
}

NWEA_ETHNICITY_MAP = {
    'african_american': 'African-American, Afro-Caribbean or Black',
    'asian_american': 'Asian-American',
    'hispanic': 'Hispanic, Latinx, or Spanish Origin',
    'middle_eastern': 'Middle Eastern or North African',
    'native_american': 'Native American or Alaska Native',
    'other': 'Not specified or Other',
    'pacific_islander': 'Native Hawaiian or Other Pacific Islander',
    'white': 'White',
    'unmatched_value': 'Not specified or Other',
    'na_value': 'Not specified or Other',
    'multiple_values': 'Multiple responses'
}

NWEA_GRADE_NAME_MAP = {
    'EC': 'EC',
    'PK': 'PK',
    'PK_3': 'PK',
    'PK_4': 'PK',
    'K': 'K',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': '10',
    '11': '11',
    '12': '12',
    'unmatched_value': None,
    'na_value': None
}

NWEA_TESTABLE_GRADES = [
    'K',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12'
]

def create_nwea_roster_and_write_locally(
    base_directory,
    filename_suffix,
    master_roster_subdirectory='master_rosters',
    master_roster_filename_stem='master_roster',
    nwea_roster_subdirectory='nwea_rosters',
    nwea_roster_filename_stem='nwea_roster'
):
    filename = os.path.join(
        base_directory,
        master_roster_subdirectory,
        '{}_{}'.format(
            master_roster_filename_stem,
            filename_suffix
        ),
        '{}_{}.pkl'.format(
            master_roster_filename_stem,
            filename_suffix
        )
    )
    master_roster_data = pd.read_pickle(filename)
    nwea_roster_data = wf_core_data.create_nwea_roster(
        master_roster_data=master_roster_data
    )
    wf_core_data.write_nwea_rosters_local(
        nwea_roster_data=nwea_roster_data,
        base_directory=base_directory,
        subdirectory=nwea_roster_subdirectory,
        filename_stem=nwea_roster_filename_stem,
        filename_suffix=filename_suffix
    )

def create_nwea_roster(
    master_roster_data
):
    ## Rename fields
    logger.info('Renaming fields')
    nwea_roster_data = (
        master_roster_data
        .rename(columns = {
            'school_name_tc': 'School Name',
            'classroom_name_tc': 'Class Name',
            'teacher_id_tc': 'Instructor ID',
            'teacher_first_name_tc': 'Instructor First Name',
            'teacher_last_name_tc': 'Instructor Last Name',
            'teacher_email_tc': 'Email Address',
            'student_id_alt_normalized_tc': 'Student State ID',
            'student_first_name_tc': 'Student First Name',
            'student_last_name_tc': 'Student Last Name',
        })
    )
    ## Create new fields
    ### User name
    logger.info('Creating user name field')
    nwea_roster_data['User Name'] = nwea_roster_data['Email Address']
    ### Student ID
    logger.info('Creating student ID field')
    nwea_roster_data['Student ID'] = nwea_roster_data.index.get_level_values('student_id_tc')
    ### Student birth date
    logger.info('Creating birth date field')
    nwea_roster_data['Student Date Of Birth'] = nwea_roster_data['student_birth_date_tc'].apply(
        lambda x: x.strftime('%m/%d/%Y')
    )
    ### Student gender
    logger.info('Creating gender field')
    nwea_roster_data['Student Gender'] = nwea_roster_data['student_gender_wf'].apply(
        lambda x: NWEA_GENDER_MAP.get(x, NWEA_GENDER_MAP.get('unmatched_value')) if pd.notna(x) else NWEA_GENDER_MAP.get('na_value')
    )
    ### Grade
    logger.info('Creating grade field')
    nwea_roster_data['Student Grade'] = nwea_roster_data['student_grade_wf'].apply(
        lambda x: NWEA_GRADE_NAME_MAP.get(x, NWEA_GRADE_NAME_MAP.get('unmatched_value')) if pd.notna(x) else NWEA_GRADE_NAME_MAP.get('na_value')
    )
    ### Student ethnicity
    logger.info('Creating ethnicity field')
    def student_race_nwea(ethnicity_list):
        if not isinstance(ethnicity_list, list):
            return NWEA_ETHNICITY_MAP.get('na_value')
        if len(ethnicity_list) > 1:
            return NWEA_ETHNICITY_MAP.get('multiple_values')
        return NWEA_ETHNICITY_MAP.get(ethnicity_list[0], NWEA_ETHNICITY_MAP.get('unmatched_value'))
    nwea_roster_data['Student Ethnic Group Name'] = nwea_roster_data['student_ethnicity_wf'].apply(student_race_nwea)
    ## Arrange columns and rows
    logger.info('Rearranging columns and rows')
    nwea_roster_data = (
        nwea_roster_data
        .reindex(columns=(
            wf_core_data.rosters.shared_constants.GROUPING_COLUMN_NAMES +
            NWEA_TARGET_COLUMN_NAMES
        ))
        .sort_values(
            wf_core_data.rosters.shared_constants.GROUPING_COLUMN_NAMES +
            ['Student Grade', 'Student First Name', 'Student Last Name']
        )
    )
    ## Create output
    logger.info('Restriction to testable grades. {} student records before restricting'.format(
        len(nwea_roster_data)
    ))
    nwea_roster_data = (
        nwea_roster_data
        .loc[nwea_roster_data['Student Grade'].isin(NWEA_TESTABLE_GRADES)]
        .copy()
        .reset_index(drop=True)
        .astype('object')
    )
    logger.info('Restricted to testable grades. {} student records after restricting'.format(
        len(nwea_roster_data)
    ))
    return nwea_roster_data

def write_nwea_rosters_local(
    nwea_roster_data,
    base_directory,
    subdirectory='nwea_rosters',
    filename_stem='nwea_roster',
    filename_suffix=None

):
    wf_core_data.rosters.shared_functions.write_rosters_local(
        roster_data=nwea_roster_data,
        base_directory=base_directory,
        subdirectory=subdirectory,
        filename_stem=filename_stem,
        filename_suffix=filename_suffix

    )
